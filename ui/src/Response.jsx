import React from 'react'
import Card from '@material-ui/core/Card'
import CardContent from '@material-ui/core/CardContent'
import ReactJson from 'react-json-view'
import { styled } from '@material-ui/core/styles'

import theme from './theme'

const LargeCard = styled(Card)({
  width: '100%',
  padding: theme.spacing(2)
})

const Response = (props) => {
  return (
    <LargeCard>
        <CardContent>
          <ReactJson
            src={props.response}
            displayObjectSize={false}
            displayDataTypes={false}
          />

        </CardContent>
    </LargeCard>
  )
}

export default Response