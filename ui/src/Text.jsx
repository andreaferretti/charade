import React from 'react'
import Button from '@material-ui/core/Button'
import Card from '@material-ui/core/Card'
import CardHeader from '@material-ui/core/CardHeader'
import CardContent from '@material-ui/core/CardContent'
import CardActions from '@material-ui/core/CardActions'
import TextField from '@material-ui/core/TextField'
import FormControlLabel from '@material-ui/core/FormControlLabel'
import FormControl from '@material-ui/core/FormControl'
import Switch from '@material-ui/core/Switch'
import Grid from '@material-ui/core/Grid'
import { styled } from '@material-ui/core/styles'

import theme from './theme'
import Languages from './Languages'
import Params from './Params'

const LargeTextField = styled(TextField)({
  width: '100%'
})

const SpacedArea = styled(CardActions)({
  padding: theme.spacing(2)
})

const Text = (props) => {
  return (
    <Card>
      <CardContent>
        <CardHeader title='Try Charade here' />
        <LargeTextField
          label='Insert text to be analyzed'
          multiline
          variant='outlined'
          rows={20}
          value={props.content}
          onChange={props.onChange}
        />
      </CardContent>
      <SpacedArea>
        <Grid container spacing={3}>
          <Grid item xs={2}>
            <FormControlLabel
              control={<Switch checked={props.debug} onChange={props.onToggleDebug} />}
              label='Debug'
              labelPlacement='start'
            />
          </Grid>
          <Grid item xs={2}>
            <Languages choices={props.langs} onChange={props.onChooseLang} lang={props.lang} />
          </Grid>
          <Grid item xs={6}>
            <Params params={props.extraParams} onChange={props.onChangeParam} />
          </Grid>
          <Grid item xs={2}>
            <FormControl>
              <Button variant='contained' color='primary' onClick={props.onSend}>Send</Button>
            </FormControl>
          </Grid>
        </Grid>

      </SpacedArea>
    </Card>
  )
}

export default Text