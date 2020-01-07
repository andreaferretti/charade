import React from 'react'
import Select from '@material-ui/core/Select'
import InputLabel from '@material-ui/core/InputLabel'
import MenuItem from '@material-ui/core/MenuItem'
import FormControl from '@material-ui/core/FormControl'

const Languages = (props) => {
  let items = [<MenuItem value={1000} key={'none'}>(none)</MenuItem>]
  var selected = 1000
  if (! props.choices) {
    return null
  }
  props.choices.forEach((lang, i) => {
    if (lang === props.lang) {
      selected = i
    }
    items.push(<MenuItem value={i} key={lang}>{lang}</MenuItem>)
  })

  return (
    <FormControl>
      <InputLabel>Language</InputLabel>
      <Select value={selected} onChange={props.onChange}>{items}</Select>
    </FormControl>
  )
}

export default Languages